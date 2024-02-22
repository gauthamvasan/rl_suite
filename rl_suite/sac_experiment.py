import argparse, os
import torch
import cv2, time, logging
logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
    
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from tqdm import tqdm
from datetime import datetime
from gymnasium.wrappers import NormalizeObservation
from rl_suite.logger import Logger
from rl_suite.mlp_policies import SquashedGaussianMLPActor, SACCritic
from rl_suite.utils import make_env, save_returns, learning_curve, save_args


class SACReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_dim, act_dim, capacity, batch_size):
        self.batch_size = batch_size
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, capacity        

        size_of_buffer = ((((self.observations.size * self.observations.itemsize) + \
                            (self.next_observations.size * self.next_observations.itemsize) + \
                            (self.actions.size * self.actions.itemsize) + (8 * capacity)) / 1024) / 1024)
        logging.info("Size of replay buffer: {:.2f}MB".format(size_of_buffer))

    def add(self, obs, action, next_obs, reward, done):        
            self.observations[self.ptr] = obs
            self.next_observations[self.ptr] = next_obs
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.dones[self.ptr] = done
            self.ptr = (self.ptr+1) % self.max_size
            self.size = min(self.size+1, self.max_size)

    def sample(self):        
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        observations = torch.from_numpy(self.observations[idxs, :])
        next_observations = torch.from_numpy(self.next_observations[idxs, :])
        actions = torch.from_numpy(self.actions[idxs])
        rewards = torch.from_numpy(self.rewards[idxs])
        dones = torch.from_numpy(self.dones[idxs])
        return (observations, actions, next_observations, rewards, dones)

    def __len__(self):
        return self.size


class SAC:
    """ SAC with Automatic Entropy Adjustment. """
    def __init__(self, cfg, device=torch.device('cpu')):
        self.cfg = cfg
        self.device = device
        self.gamma = cfg.gamma
        self.critic_tau = cfg.critic_tau
        self.encoder_tau = cfg.encoder_tau
        self.update_actor_every = cfg.update_actor_every
        self.update_critic_target_every = cfg.update_critic_target_every

        self.actor_lr = cfg.actor_lr
        self.critic_lr = cfg.critic_lr
        self.alpha_lr = cfg.alpha_lr
        self.betas = cfg.betas

        self.actor = SquashedGaussianMLPActor(cfg.obs_dim, cfg.action_dim, cfg.actor_nn_params, device)
        if cfg.use_normal_init:
            self.actor.LOG_STD_MIN = -10
            self.actor.mu.weight.data.fill_(0.0)
            self.actor.mu.bias.data.fill_(0.0)
            self.actor.log_std.weight.data.fill_(0.0)
            self.actor.log_std.bias.data.fill_(0.0)
            print('Using normal distribution initialization.')

        self.critic = SACCritic(cfg.obs_dim, cfg.action_dim, cfg.critic_nn_params, device)
        self.critic_target = deepcopy(self.critic) # also copies the encoder instance

        self.log_alpha = torch.tensor(np.log(cfg.init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod((cfg.action_dim,))

        self.num_updates = 0

        # optimizers
        self.init_optimizers()
        self.train()
        self.critic_target.train()

        # Replay buffer
        self._replay_buffer = SACReplayBuffer(cfg.obs_dim, cfg.action_dim, cfg.replay_buffer_capacity, cfg.batch_size)
        self.steps = 0

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def share_memory(self):
        self.actor.share_memory()
        self.critic.share_memory()
        self.critic_target.share_memory()
        self.log_alpha.share_memory_()

    def init_optimizers(self):
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr, betas=self.betas,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, betas=self.betas,
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.alpha_lr, betas=(0.5, 0.999),
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def sample_action(self, obs):
        with torch.no_grad():
            if not isinstance(obs, torch.FloatTensor):
                obs = torch.FloatTensor(obs).to(self.device)
                obs = obs.unsqueeze(0)
            mu, action, _, log_std = self.actor(obs)
            # print('mu:', mu.cpu().data.numpy().flatten())
            # print('std:', log_std.exp().cpu().data.numpy().flatten())
            return action.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, done):
        with torch.no_grad():
            _, policy_action, log_p, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)            
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_p
            
            if self.cfg.bootstrap_terminal:
                # enable infinite bootstrap
                target_Q = reward + (self.cfg.gamma * target_V)
            else:
                target_Q = reward + ((1.0 - done) * self.cfg.gamma * target_V)
            
        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)        
        critic_loss = torch.mean(
            (current_Q1 - target_Q) ** 2 + (current_Q2 - target_Q) ** 2
        )
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        critic_stats = {
            'train/critic_loss': critic_loss.item()
        }

        return critic_stats

    def update_actor_and_alpha(self, obs):
        # detach encoder, so we don't update it with the actor loss
        _, action, log_p, log_std = self.actor(obs)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_p - actor_Q).mean()

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)        

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_p - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        actor_stats = {
            'train/actor_loss': actor_loss.item(),
            'train_actor/target_entropy': self.target_entropy.item(),
            'train/ent_loss': alpha_loss.item(),
            'train_alpha/value': self.alpha.item(),
            'train/entropy': entropy.mean().item(),
        }
        return actor_stats

    def update(self, obs, action, next_obs, reward, done):
        # Move tensors to device
        obs, action, reward, next_obs, done = obs.to(self.device), action.to(self.device), \
            reward.to(self.device), next_obs.to(self.device), done.to(self.device)

        # print(obs.shape, action.shape, reward.shape, next_obs.shape, done.shape)
        # regular update of SAC_RAD, sequentially augment data and train
        stats = self.update_critic(obs, action, reward, next_obs, done)
        if self.num_updates % self.update_actor_every == 0:
            actor_stats = self.update_actor_and_alpha(obs)
            stats = {**stats, **actor_stats}
        if self.num_updates % self.update_critic_target_every == 0:
            self.soft_update_target()
        stats['train/batch_reward'] = reward.mean().item()
        stats['train/num_updates'] = self.num_updates
        self.num_updates += 1
        return stats

    @staticmethod
    def soft_update_params(net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

    def soft_update_target(self):
        self.soft_update_params(
            self.critic.Q1, self.critic_target.Q1, self.critic_tau
        )
        self.soft_update_params(
            self.critic.Q2, self.critic_target.Q2, self.critic_tau
        )
    
    def save(self, model_dir, unique_str):
        model_dict = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_alpha": self.log_alpha.detach().item(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
            "log_alpha_opt": self.log_alpha_optimizer.state_dict(),
        }
        torch.save(
            model_dict, '%s/%s.pt' % (model_dir, unique_str)
        )

    def load(self, model_dir, unique_str):
        model_dict = torch.load('%s/%s.pt' % (model_dir, unique_str))
        self.actor.load_state_dict(model_dict["actor"])
        self.critic.load_state_dict(model_dict["critic"])
        self.log_alpha = torch.tensor(model_dict["log_alpha"]).to(self.device)
        self.log_alpha.requires_grad = True
        self.actor_optimizer.load_state_dict(model_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(model_dict["critic_optimizer"])
        self.log_alpha_optimizer.load_state_dict(model_dict["log_alpha_optimizer"])

    def push_and_update(self, obs, action, next_obs, reward, done):
        self._replay_buffer.add(obs, action, next_obs, reward, done)
        self.steps += 1
        
        stat = {}
        if self.steps > self.cfg.init_steps and (self.steps % self.cfg.update_every == 0):
            for _ in range(self.cfg.update_epochs):
                tic = time.time()
                stat = self.update(*self._replay_buffer.sample())
                # if self.num_updates %100 == 0:
                    # print(f"Update {self.num_updates} took {time.time() - tic}s")
        return stat
       
     
def main(args):    
    start_time = datetime.now()

    #### Unique IDs
    run_id = datetime.now().strftime("%Y-%m-%d-%H_%M_%S") + f"_seed-{args.seed}"
    lc_path = f"{args.results_dir}/{run_id}_learning_curve.png"
    rets_path = f"{args.results_dir}/{run_id}_returns.txt"
    L = Logger(args.results_dir, prefix=f"{run_id}_", use_tb=False)
    os.makedirs(args.results_dir, exist_ok=True)
    ####

    #### Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    ####

    # Env
    env = make_env(args)
    if args.normalize_obs:
        env = NormalizeObservation(env)

    # Agent    
    args.obs_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    agent = SAC(cfg=args, device=torch.device(args.device))

    # Experiment block starts
    experiment_done = False
    total_steps = 0
    returns, ep_lens = [], []
    logging.info(f'Experiment starts at: {start_time}')
    while not experiment_done:
        obs, _ = env.reset() # start a new episode
        ret, sub_epi, epi_steps, sub_steps, terminated = 0, 0, 0, 0, False
        epi_start_time = time.time()
        while not terminated:
            # Select an action
            action = agent.sample_action(obs)

            # Observe
            next_obs, r, terminated, truncated, _ = env.step(action)
            
            # Learn
            stat = agent.push_and_update(obs, action, next_obs, r, terminated)

            for k, v in stat.items():
                L.log(k, v, total_steps)

            # if total_steps % 50 == 0: 
            #     print("Step: {}, Next obs: {}, reward: {}, done: {}".format(total_steps, next_obs[:2], r, terminated))

            obs = next_obs

            # Log
            total_steps += 1
            ret += r
            epi_steps += 1
            sub_steps += 1

            # Save model
            if args.model_checkpoint:
                if total_steps % args.model_checkpoint == 0:
                    agent.save(model_dir=args.results_dir, unique_str=f"{run_id}_model")
            
            if not terminated and truncated: # set timeout here
                sub_steps = 0
                sub_epi += 1

                # Prevent discontinuity in saving models
                x = total_steps//10000
                y = (total_steps + args.reset_penalty_steps)//10000
                # Save model
                if args.model_checkpoint and x != y:
                    agent.save(model_dir=args.results_dir, unique_str=f"{run_id}_model")

                ret += args.reset_penalty_steps * args.reward
                epi_steps += args.reset_penalty_steps
                total_steps += args.reset_penalty_steps
                # logging.info(f'Sub episode {sub_epi} done. Total steps: {total_steps}')
                if args.env in ["dm_reacher_easy", "dm_reacher_hard", "point_maze"]:
                    obs, _ = env.reset(randomize_target=truncated)
                else:
                    obs, _ = env.reset()

            if total_steps >= args.N:
                experiment_done = True
                break

        # if done: # episode done, save result
        L.log('train/duration', time.time() - epi_start_time, total_steps)
        L.log('train/episode_return', ret, total_steps)
        L.log('train/sub_episode', sub_epi, total_steps)
        L.log('train/episode', len(returns), total_steps)
        L.dump(total_steps)

        returns.append(ret)
        ep_lens.append(epi_steps)
        save_returns(ep_lens=ep_lens, rets=returns, save_path=rets_path)
        learning_curve(rets=returns, ep_lens=ep_lens, save_path=lc_path)
        # logging.info(f"Episode {len(returns)} ended after {epi_steps} steps with return {ret:.2f}. Total steps: {total_steps}")

    duration = datetime.now() - start_time    
    agent.save(model_dir=args.results_dir, unique_str=f"{run_id}_model")
    save_args(args, f"{args.results_dir}/{run_id}_args.json")
    save_returns(ep_lens=ep_lens, rets=returns, save_path=rets_path)
    learning_curve(rets=returns, ep_lens=ep_lens, save_path=lc_path)    
    logging.info(f"Finished in {duration}")
    return ep_lens, returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Task
    parser.add_argument('--env', required=True, type=str, help="e.g., 'ball_in_cup', 'dm_reacher_easy', 'dm_reacher_hard', 'Hopper-v2' ")        
    parser.add_argument('--seed', required=True, type=int, help="Seed for random number generator")       
    parser.add_argument('--N', required=True, type=int, help="# timesteps for the run")
    parser.add_argument('--timeout', required=True, type=int, help="Timeout for the env")        
    ## Minimum-time tasks
    parser.add_argument('--reward', default=-1, type=float, help="Reward penalty for min-time specification")
    parser.add_argument('--reset_penalty_steps', default=20, type=float, help="Reset penalty steps for min-time specification")
    ## Mujoco sparse reacher
    parser.add_argument('--tol', default=0.009, type=float, help="Target size in [0.09, 0.018, 0.036, 0.072]")
    ## DotReacher
    parser.add_argument('--pos_tol', default=0.1, type=float, help="Position tolerance in [0.05, ..., 0.25]")
    parser.add_argument('--vel_tol', default=0.05, type=float, help="Velocity tolerance in [0.05, ..., 0.1]")
    parser.add_argument('--dt', default=0.2, type=float, help="Simulation action cycle time")
    parser.add_argument('--clamp_action', default=1, type=int, help="Clamp action space")    
    ## Point Maze
    parser.add_argument('--maze_type', default="small", type=str, help= "Maze type in ['small', 'medium', 'large']")
    parser.add_argument('--reward_type', default="sparse", type=str, help= "Reward type in ['sparse', 'dense']")
    # Algorithm        
    parser.add_argument('--replay_buffer_capacity', required=True, type=int)
    parser.add_argument('--init_steps', required=True, type=int)
    parser.add_argument('--update_every', default=2, type=int)
    parser.add_argument('--update_epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor")
    parser.add_argument('--bootstrap_terminal', default=0, type=int, help="Bootstrap on terminal state")
    parser.add_argument('--normalize_obs', action="store_true")
    parser.add_argument('--betas', default="0.9 0.999", type=str)
    ## Actor
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--update_actor_every', default=1, type=int)
    ## Critic
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--update_critic_target_every', default=1, type=int)
    ## Entropy
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=3e-4, type=float)
    ## Encoder
    parser.add_argument('--encoder_tau', default=0.005, type=float)
    parser.add_argument('--l2_reg', default=1e-4, type=float, help="L2 regularization coefficient")        
    # MLP params
    parser.add_argument('--actor_hidden_sizes', default="256,256", type=str)
    parser.add_argument('--critic_hidden_sizes', default="256,256", type=str)
    parser.add_argument('--nn_activation', default="relu", type=str)
    parser.add_argument('--use_normal_init', action='store_true')
    # Misc
    parser.add_argument('--init_policy_test', action='store_true', help="Initiate hits vs timeout test")
    parser.add_argument('--results_dir', required=True, type=str, help="Save results to this dir")        
    parser.add_argument('--checkpoint', default=5000, type=int, help="Save plots and rets every checkpoint")
    parser.add_argument('--model_checkpoint', default=0, type=int, help="Save plots and rets every checkpoint")
    parser.add_argument('--device', default="cuda", type=str)
    args = parser.parse_args()
    
    assert args.reward < 0 and args.reset_penalty_steps >= 0    
    args.betas = list(map(float, args.betas.split()))

    args.actor_nn_params = {
        'mlp': {
            'hidden_sizes': list(map(int, args.actor_hidden_sizes.split(","))),
            'activation': args.nn_activation,
        }
    }
    args.critic_nn_params = {
        'mlp': {
            'hidden_sizes': list(map(int, args.critic_hidden_sizes.split(","))),
            'activation': args.nn_activation,
        }
    }

    main(args)
