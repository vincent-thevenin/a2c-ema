import torch.nn as nn
import torch
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, num_actions, num_inputs):
        super(Actor, self).__init__()
        
        self.lin1 = nn.Linear(num_inputs, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, num_actions)
    
    def forward(self, s):
        out = self.lin1(s)
        out = torch.tanh(out)
        out = self.lin2(out)
        out = torch.tanh(out)
        out = self.lin3(out)
        
        return out

class Q(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(Q, self).__init__()

        self.lin1 = nn.Linear(num_inputs + num_actions, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, 1)

    def forward(self, s, a):
        out = self.lin1(torch.cat((s, a), dim=-1))
        out = torch.tanh(out)
        out = self.lin2(out)
        out = torch.tanh(out)
        out = self.lin3(out)

        return out

class V(nn.Module):
    def __init__(self, num_inputs):
        super(V, self).__init__()

        self.lin1 = nn.Linear(num_inputs, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, 1)

    def forward(self, s):
        out = self.lin1(s)
        out = torch.tanh(out)
        out = self.lin2(out)
        out = torch.tanh(out)
        out = self.lin3(out)

        return out