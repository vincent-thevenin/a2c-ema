from ast import Num
import torch.nn as nn
import torch
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, num_actions, num_inputs):
        super(Actor, self).__init__()
        
        self.lin1 = nn.Linear(num_inputs, (num_inputs + num_actions) * 2)
        self.lin2 = nn.Linear((num_inputs + num_actions) * 2, num_actions)
    
    def forward(self, s):
        out = self.lin1(s)
        out = torch.tanh(out)
        out = self.lin2(out)
        
        return out
    
class Q(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(Q, self).__init__()

        self.lin1 = nn.Linear(num_inputs + num_actions, (num_inputs + num_actions) * 2)
        self.lin2 = nn.Linear((num_inputs + num_actions) * 2, 1)

    def forward(self, s, a):
        out = self.lin1(torch.cat((s,a), dim=-1))
        out = torch.tanh(out)
        out = self.lin2(out)

        return out