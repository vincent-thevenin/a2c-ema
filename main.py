import gym
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pickle
from scipy.ndimage import uniform_filter1d
from datetime import datetime
import torch
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.utils import tensorboard
from tqdm import tqdm

from dataset import ReplayQueue, ModelDataset
from model import Actor, Q


def argmax_logits2onehot(logits):
        action_sample = torch.zeros_like(logits)
        action_sample.scatter_(-1, logits.argmax(-1, keepdim=True), 1)
        return action_sample

class Norms:
    def __init__(self, env_name):
        self.means = {
            'Acrobot-v1': torch.Tensor(
                [1, 1, 1, 1, 12.57, 28.27]
            ),
        }

        self.means = self.means[env_name] if env_name in self.means else 1

    def __call__(self, x):
        return x / self.means

def sim_ema(
    gamma: float,
    num_steps: int,
    eval_interval: int,
    ema_recall_interval: int,
    lr: float,
    env_name: str,
):
    env = gym.make(env_name)
    env_eval = gym.make(env_name)
    norms = Norms(env_name)

    actor = Actor(env.action_space.n, env.observation_space.shape[0])
    actor_ema = Actor(env.action_space.n, env.observation_space.shape[0])
    with torch.no_grad():
        actor_ema.load_state_dict(actor.state_dict())
    q = Q(env.action_space.n, env.observation_space.shape[0])
    q_ema = Q(env.action_space.n, env.observation_space.shape[0])
    with torch.no_grad():
        q_ema.load_state_dict(q.state_dict())

    optim_actor = torch.optim.SGD(actor.parameters(), lr=lr)
    optim_q = torch.optim.SGD(q.parameters(), lr=lr)
    optim_q_ema = torch.optim.SGD(q_ema.parameters(), lr=lr)

    replay = ReplayQueue(
        capacity,
        use_prioritized_replay,
    )
    # preheat data
    with torch.no_grad():
        s = env.reset()
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        s = norms(s)
        a = actor(s) * 0
        for i in range(max(1, int(0.1 * capacity))):
            a_sample = OneHotCategorical(logits=a).sample()
            s_new, r, done, _ = env.step(a_sample.argmax().item())
            s_new = torch.tensor(s_new, dtype=torch.float32).unsqueeze(0)
            s_new = norms(s_new)
            target = r + gamma * q_ema(s_new, argmax_logits2onehot(actor_ema(s_new))) * (1 - done)
            value = q(s, a_sample)
            error = torch.nn.functional.mse_loss(
                target,
                value,
            )
            replay.push((s, a_sample, r, s_new, done), error.item())
            s = s_new
            s = s_new
            if done:
                s = env.reset()
                s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                s = norms(s)
    ds = ModelDataset(replay)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    s_env = env.reset()
    s_env = torch.tensor(s_env, dtype=torch.float32).unsqueeze(0)
    s_env = norms(s_env)
    pbar = tqdm(range(int(num_steps)))
    cum_r_eval_list_ema = []
    steps = 0
    # for i in pbar:
    while steps < num_steps:
        for s, a, r, s_new, done, idx in dl:
            with torch.no_grad():
                # Simulate one step
                a_env = actor_ema(s_env)
                a_dist = OneHotCategorical(logits=a_env)
                a_env_sample = a_dist.sample()

                s_env_new, r_env, done_env, _ = env.step(a_env_sample.argmax().item())
                s_env_new = torch.tensor(s_env_new, dtype=torch.float32).unsqueeze(0)
                s_env_new = norms(s_env_new)

                # # Get replay queue values
                # a = actor_ema(s)
                # a_dist = OneHotCategorical(logits=a)
                # a_sample = a_dist.sample()

                # update critic
                target = r + gamma * q_ema( # TODO TEST q_ema here? like q target?
                    s_new,
                    argmax_logits2onehot(actor_ema(s_new))
                ) * (1 - done)
            loss_q = 1/2 * (target - q(s, a)) ** 2
            loss_q.mean().backward()
            optim_q.step()
            optim_q.zero_grad()

            # update replay
            replay.update_priorities(idx, loss_q)
            replay.push((s_env, a_env_sample, r_env, s_env_new, done_env), 1)
            s_env = s_env_new
            if done_env:
                s_env = env.reset()
                s_env = torch.tensor(s_env, dtype=torch.float32).unsqueeze(0)
                s_env = norms(s_env)
            steps += 1

            # update critic_ema with exponential moving average
            for param, ema_param in zip(q.parameters(), q_ema.parameters()):
                ema_param.data = param.data * (1 - eps_ema) + ema_param.data * eps_ema

            # update actor
            a = actor(s)
            a_dist = OneHotCategorical(logits=a)
            a_sample = a_dist.sample()
            a_prob = torch.nn.functional.softmax(a, dim=-1)
            loss_actor = -q_ema(s, a_sample + a_prob - a_prob.detach()).mean()
            # loss_actor = -(a_dist.log_prob(a_sample) * (target - q_ema(s, a_sample).detach())).mean()
            # loss_actor -= max(0.001, 10 * (num_steps - steps*2) / num_steps) * a_dist.entropy().mean()
            loss_actor.backward()
            optim_actor.step()
            optim_actor.zero_grad()
            optim_q.zero_grad()
            optim_q_ema.zero_grad()

            # update actor_ema with exponential moving average
            for param, ema_param in zip(actor.parameters(), actor_ema.parameters()):
                ema_param.data = param.data * (1 - eps_ema) + ema_param.data * eps_ema

            if steps % ema_recall_interval == 0:
                with torch.no_grad():
                    actor.load_state_dict(actor_ema.state_dict())
                    q.load_state_dict(q_ema.state_dict())

            if steps % eval_interval == 0:
                #eval
                with torch.no_grad():
                    cum_r_eval = 0
                    s_eval = env_eval.reset()
                    s_eval = torch.tensor(s_eval, dtype=torch.float32).unsqueeze(0)
                    s_eval = norms(s_eval)
                    done_eval = False
                    while not done_eval:
                        a_eval = actor_ema(s_eval)
                        a_sample_eval = argmax_logits2onehot(a_eval)
                        s_eval_new, r_eval, done_eval, _ = env_eval.step(a_sample_eval.argmax().item())
                        s_eval_new = torch.tensor(s_eval_new, dtype=torch.float32).unsqueeze(0)
                        s_eval_new = norms(s_eval_new)
                        cum_r_eval += r_eval
                        s_eval = s_eval_new

                    pbar.set_postfix(cum_r_eval=cum_r_eval)
                    cum_r_eval_list_ema.append(cum_r_eval)
            
            pbar.update(1)
            if steps >= num_steps:
                break

    return cum_r_eval_list_ema


def sim(
    gamma: float,
    num_steps: int,
    eval_interval: int,
    ema_recall_interval: int,
    lr: float,
    env_name: str,
):
    env = gym.make(env_name)
    env_eval = gym.make(env_name)
    norms = Norms(env_name)

    actor = Actor(env.action_space.n, env.observation_space.shape[0])
    q = Q(env.action_space.n, env.observation_space.shape[0])
    optim_actor = torch.optim.SGD(actor.parameters(), lr=lr)
    optim_q = torch.optim.SGD(q.parameters(), lr=lr)
    s = env.reset()
    s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
    s = norms(s)
    pbar = tqdm(range(int(num_steps)))
    cum_r_eval_list = []
    for i in pbar:
        # env.render()
        a = actor(s)

        a_dist = OneHotCategorical(logits=a)
        a_sample = a_dist.sample()

        s_new, r, done, _ = env.step(a_sample.argmax().item())
        s_new = torch.tensor(s_new, dtype=torch.float32).unsqueeze(0)
        s_new = norms(s_new)

        # update critic
        with torch.no_grad():
            target = r + gamma * q(
                s_new,
                argmax_logits2onehot(actor(s_new))
            ) * (1 - done)
        q_sample = q(s, a_sample)
        loss_q = torch.nn.functional.mse_loss(
            q_sample,
            target
        )
        loss_q.backward()
        optim_q.step()
        optim_q.zero_grad()

        # update actor
        a_prob = torch.nn.functional.softmax(a, dim=-1)
        loss_actor = -q(s, a_sample + a_prob - a_prob.detach()).mean()
        # loss_actor = -(a_dist.log_prob(a_sample) * (target - q_sample.detach())).mean()
        # loss_actor -= max(0.001, 10 * (num_steps - i*2) / num_steps) * a_dist.entropy().mean()
        loss_actor.backward()
        optim_actor.step()
        optim_actor.zero_grad()
        optim_q.zero_grad()

        s = s_new
        if done:
            s = env.reset()
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            s = norms(s)

        if i % eval_interval == 0:
            #eval
            with torch.no_grad():
                cum_r_eval = 0
                s_eval = env_eval.reset()
                s_eval = torch.tensor(s_eval, dtype=torch.float32).unsqueeze(0)
                done_eval = False
                while not done_eval:
                    a_eval = actor(s_eval)
                    a_sample_eval = argmax_logits2onehot(a_eval)
                    s_eval_new, r_eval, done_eval, _ = env_eval.step(a_sample_eval.argmax().item())
                    s_eval_new = torch.tensor(s_eval_new, dtype=torch.float32).unsqueeze(0)
                    s_eval_new = norms(s_eval_new)
                    cum_r_eval += r_eval
                    s_eval = s_eval_new

                pbar.set_postfix(cum_r_eval=cum_r_eval)
                cum_r_eval_list.append(cum_r_eval)

    return cum_r_eval_list


if __name__ == '__main__':
    batch_size = 64
    gamma = 0.95
    eps_ema = 0.99 # EMA: param * (1-eps) + ema_param * eps
    num_steps = 100_000
    eval_interval = 100
    ema_recall_interval = num_steps // 1000
    lr = 0.01
    env_name = 'CartPole-v1' #'Acrobot-v1'
    num_experiments = 50
    filter_n = 10
    capacity = batch_size
    use_prioritized_replay = False

    print(
        f'batch_size: {batch_size}, gamma: {gamma}, eps_ema: {eps_ema}, num_steps: {num_steps}, eval_interval: {eval_interval}, ema_recall_interval: {ema_recall_interval}, lr: {lr}, env_name: {env_name}, num_experiments: {num_experiments}, filter_n: {filter_n}, capacity: {capacity}'
    )

    eval_list_ema = []

    eval_list = []

    # sim_ema(
    #     gamma,
    #     num_steps,
    #     eval_interval,
    #     ema_recall_interval,
    #     lr,
    #     env_name
    # )
    with mp.Pool(min(mp.cpu_count(), num_experiments)) as p:
        result_ema = p.starmap(sim_ema, [(
            gamma,
            num_steps,
            eval_interval,
            ema_recall_interval,
            lr,
            env_name
        )]*num_experiments)
    result_ema = np.array(result_ema)
    ema_mean = np.mean(result_ema, axis=0)
    ema_std = np.std(result_ema, axis=0)

    tensorboard_writer = tensorboard.SummaryWriter(
        comment=f'eps_{eps_ema}_recall_{ema_recall_interval}_lr_{lr}_env_{env_name}'
    )
    for i in range(len(ema_mean)):
        tensorboard_writer.add_scalar('mean', ema_mean[i], i)
        tensorboard_writer.add_scalar('std', ema_std[i], i)
    tensorboard_writer.close()

    # ema_mean = uniform_filter1d(ema_mean, size=filter_n)
    # ema_std = uniform_filter1d(ema_std, size=filter_n)

    # plt.plot(ema_mean, label='actor_critic_ema')
    # plt.fill_between(range(len(ema_mean)), ema_mean - ema_std, ema_mean + ema_std, alpha=0.2)
    # plt.savefig('cartpole_actor_critic_ema.png')
    # plt.clf()

    exit()
    with mp.Pool(mp.cpu_count()) as p:
        result = p.starmap(sim, [(
            gamma,
            num_steps,
            eval_interval,
            ema_recall_interval,
            lr,
            env_name
        )]*num_experiments)

    result = np.array(result)

    ac_mean = np.mean(result, axis=0)
    ac_std = np.std(result, axis=0)

    tensorboard_writer = tensorboard.SummaryWriter(
        comment=f'lr_{lr}_env_{env_name}'
    )
    for i in range(len(ac_mean)):
        tensorboard_writer.add_scalar('mean', ac_mean[i], i)
        tensorboard_writer.add_scalar('std',  ac_std[i], i)
    tensorboard_writer.close()

    # ac_mean = uniform_filter1d(ac_mean, size=filter_n)
    # ac_std = uniform_filter1d(ac_std, size=filter_n)

    # plt.plot(ac_mean, label='actor_critic')
    # plt.fill_between(range(len(ac_mean)), ac_mean - ac_std, ac_mean + ac_std, alpha=0.2)
    # plt.savefig('cartpole_actor_critic_ac.png')
    # plt.clf()

    # plt.plot(ema_mean, label='actor_critic_ema')
    # plt.fill_between(range(len(ema_mean)), ema_mean - ema_std, ema_mean + ema_std, alpha=0.2)
    # plt.plot(ac_mean, label='actor_critic')
    # plt.fill_between(range(len(ac_mean)), ac_mean - ac_std, ac_mean + ac_std, alpha=0.2)
    # plt.savefig('cartpole_actor_critic_ac_ema.png')
