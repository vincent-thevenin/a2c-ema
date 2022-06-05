import numpy as np
import pickle
import torch
from torch.utils.data import Dataset


class ReplayQueue():
    def __init__(self, capacity):
        self.capacity = capacity

        self.errors = []
        self.memory = []
        self.probs = []

    def push(self, sarst, error):
        if len(self.memory) >= self.capacity:
            self.errors.pop(0)
            self.memory.pop(0)

        self.errors.append(error)
        self.memory.append(sarst)
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
        for idx, error in zip(idxs, errors):
            self.errors[idx] = error.item()
        self.probs = np.exp(self.errors) / sum(np.exp(self.errors))

    def __len__(self):
        # retrieve number of steps
        return len(self.memory)

class ModelDataset(Dataset):
    def __init__(self, replay: ReplayQueue):
        self.replay = replay  # replay is updated outside
        
    def __len__(self):
        return len(self.replay)

    def __getitem__(self, idx):
        idx = np.random.choice(
            len(self.replay),
            p = self.replay.probs
        )

        state, action, reward, next_state, is_terminal = self.replay.memory[idx]

        reward = torch.Tensor([reward])
        is_terminal = torch.Tensor([float(is_terminal)])

        return state, action, reward.unsqueeze(1), next_state, is_terminal.unsqueeze(1), idx