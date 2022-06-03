import pickle
import torch
from torch.utils.data import Dataset


class ReplayQueue():
    def __init__(self, capacity):
        self.capacity = capacity

        self.memory = []

    def push(self, sarst):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)

        self.memory.append(sarst)

    def load(self, path):
        with open(path, 'rb') as f:
            self.memory = pickle.load(f)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.memory, f)

    def clear(self):
        self.memory = []

    def __len__(self):
        # retrieve number of steps
        return len(self.memory)

class ModelDataset(Dataset):
    def __init__(self, replay: ReplayQueue):
        self.replay = replay  # replay is updated outside
        
    def __len__(self):
        return len(self.replay)

    def __getitem__(self, idx):
        state, action, reward, next_state, is_terminal = self.replay.memory[idx]

        reward = torch.Tensor([reward])
        is_terminal = torch.Tensor([float(is_terminal)])

        return state, action, reward.unsqueeze(1), next_state, is_terminal.unsqueeze(1)