import random, logging, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim, vizdoom as vzd, math, gc, psutil, os
from dataclasses import dataclass, asdict
from collections import deque
from datetime import datetime
from typing import Optional

@dataclass
class TrainingConfig:
    map_name: str = "ROOM"
    frame_skip: int = 4
    episodes: int = 3000
    learning_rate: float = 3e-4
    lr_decay: float = 0.9999
    gamma: float = 0.99
    replay_buffer_size: int = 100_000
    min_replay_size: int = 10_000
    batch_size: int = 64
    n_steps: int = 3
    target_update_freq: int = 2000
    train_freq: int = 4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 500_000
    epsilon_power: float = 1.5
    use_noisy_nets: bool = True
    use_double_dqn: bool = True
    clip_rewards: bool = True
    normalize_advantages: bool = True
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gc_interval: int = 100
    clear_cuda_cache_interval: int = 50
    memory_warning_threshold: float = 80.0
    gradient_clip_value: float = 10.0
    n_stack_frames: int = 1
    use_depth: bool = True
    screen_format: int = 8
    crosshair: bool = True
    hud: str = "none"
    
    def to_json_dict(self):
        return {
            "algo_type": "QVALUE",
            "n_stack_frames": self.n_stack_frames,
            "extra_state": ["depth"] if self.use_depth else None,
            "hud": self.hud,
            "crosshair": self.crosshair,
            "screen_format": self.screen_format
        }

class MemoryMonitor:
    def __init__(self, logger, warning_threshold=80.0):
        self.logger = logger
        self.warning_threshold = warning_threshold
        self.last_warning_time = 0
        
    def check_memory(self):
        memory_percent = psutil.virtual_memory().percent
        
        if memory_percent > self.warning_threshold:
            current_time = datetime.now().timestamp()
            if current_time - self.last_warning_time > 60:
                self.logger.warning(f"High memory usage: {memory_percent:.1f}%")
                self.last_warning_time = current_time
                
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                if allocated > 0:
                    self.logger.debug(f"GPU {i} - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
        
        return memory_percent

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
        
    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
        
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
        
    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, 
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))
        return x * attention

class DuelingDQN(nn.Module):
    def __init__(self, input_channels: int, action_space_size: int, use_noisy: bool = True):
        super().__init__()
        self.use_noisy = use_noisy
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.attention1 = SpatialAttention(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.attention2 = SpatialAttention(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 128, 128)
            conv_out = self._forward_conv(dummy)
            conv_out_size = conv_out.shape[1]
            
        if use_noisy:
            self.value_stream = nn.Sequential(
                NoisyLinear(conv_out_size, 512),
                nn.ReLU(),
                NoisyLinear(512, 1)
            )
            self.advantage_stream = nn.Sequential(
                NoisyLinear(conv_out_size, 512),
                nn.ReLU(),
                NoisyLinear(512, action_space_size)
            )
        else:
            self.value_stream = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, action_space_size)
            )
            
    def _forward_conv(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.attention1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.attention2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        return x.flatten(1)
        
    def forward(self, x):
        features = self._forward_conv(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
        
    def reset_noise(self):
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

class CompressedExperience:
    def __init__(self, state, action, reward, next_state, done):
        self.state = (state.cpu() * 255).to(torch.uint8)
        self.action = action
        self.reward = np.float16(reward)
        self.next_state = (next_state.cpu() * 255).to(torch.uint8)
        self.done = done
        
    def decompress(self, device):
        state = self.state.to(device, non_blocking=True).float() / 255.0
        next_state = self.next_state.to(device, non_blocking=True).float() / 255.0
        return state, self.action, float(self.reward), next_state, self.done

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = [None] * capacity
        self.n_entries = 0
        self.write = 0
        
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
            
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
            
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
            
    def total(self):
        return self.tree[0]
        
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1
            
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
        
    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, n_steps, gamma, alpha=0.6):
        self.n_step_buffer = deque(maxlen=n_steps)
        self.tree = SumTree(capacity)
        self.n_steps = n_steps
        self.gamma = gamma
        self.alpha = alpha
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
        self.abs_err_upper = 1.0
        self.min_priority = 0.01
        
    def _get_n_step_info(self):
        R = 0
        for i in range(len(self.n_step_buffer)):
            R += (self.gamma ** i) * self.n_step_buffer[i][2]
            
        state, action, _, _ = self.n_step_buffer[0]
        next_state, _, _, done = self.n_step_buffer[-1]
        
        return state, action, R, next_state, done
        
    def push(self, state, action, reward, done):
        if done and len(self.n_step_buffer) > 0:
            while len(self.n_step_buffer) > 0:
                state_0, action_0, reward_0, _ = self.n_step_buffer[0]
                
                R = 0
                for i in range(len(self.n_step_buffer)):
                    R += (self.gamma ** i) * self.n_step_buffer[i][2]
                
                experience = CompressedExperience(state_0, action_0, R, state, True)
                
                max_p = np.max(self.tree.tree[-self.tree.capacity:])
                if max_p == 0:
                    max_p = self.abs_err_upper
                self.tree.add(max_p, experience)
                
                self.n_step_buffer.popleft()
            
        self.n_step_buffer.append((state, action, reward, done))
        
        if len(self.n_step_buffer) < self.n_steps and not done:
            return
            
        state, action, R, next_state, done = self._get_n_step_info()
        
        experience = CompressedExperience(state, action, R, next_state, done)
        
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
            
        self.tree.add(max_p, experience)
        
    def sample(self, batch_size):
        b_idx = np.empty((batch_size,), dtype=np.int32)
        b_memory = []
        ISWeights = np.empty((batch_size, 1), dtype=np.float32)
        
        pri_seg = self.tree.total() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        
        valid_samples = 0
        for i in range(batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            
            if data is None:
                continue
                
            prob = p / self.tree.total()
            ISWeights[valid_samples, 0] = np.power(self.tree.n_entries * prob, -self.beta)
            b_idx[valid_samples] = idx
            b_memory.append(data)
            valid_samples += 1
            
        if valid_samples == 0:
            return None, None, None
            
        b_idx = b_idx[:valid_samples]
        ISWeights = ISWeights[:valid_samples]
        ISWeights /= ISWeights.max()
        
        return b_idx, b_memory, torch.from_numpy(ISWeights).float()
        
    def update_priorities(self, tree_idx, abs_errors):
        abs_errors += self.min_priority
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
            
    def __len__(self):
        return self.tree.n_entries

def create_doom_game(config):
    game = vzd.DoomGame()
    game.set_doom_scenario_path("jku.wad")
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_resolution(vzd.ScreenResolution.RES_256X192)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    
    buttons = [
        vzd.Button.MOVE_FORWARD,
        vzd.Button.ATTACK,
        vzd.Button.MOVE_LEFT,
        vzd.Button.MOVE_RIGHT,
        vzd.Button.TURN_LEFT,
        vzd.Button.TURN_RIGHT,
        vzd.Button.JUMP,
    ]
    game.set_available_buttons(buttons)
    
    variables = [
        vzd.GameVariable.FRAGCOUNT,
        vzd.GameVariable.DEATHCOUNT,
        vzd.GameVariable.HITCOUNT,
        vzd.GameVariable.HITS_TAKEN
    ]
    game.set_available_game_variables(variables)
    
    game.set_render_hud(config.hud == "full")
    game.set_render_crosshair(config.crosshair)
    game.set_depth_buffer_enabled(config.use_depth)
    game.set_window_visible(False)
    game.set_episode_timeout(2000)
    
    game.add_game_args(
        f"-host 1 -deathmatch +viz_bots_path easy_bots.cfg "
        f"+sv_forcerespawn 1 +sv_respawnprotect 1 +viz_respawn_delay 2 "
        f"+sv_nocrouch 1 +name AI-Agent +colorset 0"
    )
    
    game.set_doom_map("map01")
    game.init()
    
    for _ in range(4):
        game.send_game_command("addbot")
        
    return game

def create_action_space():
    return [
        [False, False, False, False, False, False, False],
        [True, False, False, False, False, False, False],
        [False, True, False, False, False, False, False],
        [False, False, True, False, False, False, False],
        [False, False, False, True, False, False, False],
        [False, False, False, False, True, False, False],
        [False, False, False, False, False, True, False],
        [False, False, False, False, False, False, True],
    ]

def preprocess_frame(game_state, device, config) -> torch.Tensor:
    if game_state is None:
        channels = 1 + (1 if config.use_depth else 0)
        return torch.zeros((channels, 128, 128), dtype=torch.float32, device=device)
        
    with torch.no_grad():
        screen = torch.from_numpy(game_state.screen_buffer).to(device, non_blocking=True).float()
        if screen.dim() == 2:
            screen = screen.unsqueeze(0)
        
        obs = {"screen": screen}
        
        if config.use_depth and game_state.depth_buffer is not None:
            depth = torch.from_numpy(game_state.depth_buffer).to(device, non_blocking=True).float()
            if depth.dim() == 2:
                depth = depth.unsqueeze(0)
            obs["depth"] = depth
        
        for key in obs:
            buffer = obs[key]
            buffer = buffer.unsqueeze(0)
            buffer = F.interpolate(buffer, (128, 128), mode='bilinear', align_corners=False)
            obs[key] = buffer.squeeze(0)
        
        obs["screen"] = obs["screen"] / 255.0
        if "depth" in obs:
            obs["depth"] = obs["depth"] / (obs["depth"].max() + 1e-8)
            obs["depth"] = torch.nan_to_num(obs["depth"])
        
        buffers = ["screen", "depth"] if "depth" in obs else ["screen"]
        result = torch.cat([obs[k] for k in buffers], dim=0)
        
    return result

class Agent:
    def __init__(self, config: TrainingConfig, action_space_size: int, logger):
        self.config = config
        self.action_space_size = action_space_size
        self.logger = logger
        self.epsilon = config.epsilon_start
        self.total_steps = 0
        
        self.input_channels = 1
        if config.use_depth:
            self.input_channels += 1
        
        self.memory = PrioritizedReplayBuffer(
            config.replay_buffer_size, 
            config.n_steps, 
            config.gamma
        )
        
        self.policy_net = DuelingDQN(
            self.input_channels, action_space_size, config.use_noisy_nets
        ).to(config.device)
        
        self.target_net = DuelingDQN(
            self.input_channels, action_space_size, config.use_noisy_nets
        ).to(config.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), 
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, 
            gamma=config.lr_decay
        )
        
        self.best_model_state_dict = self.policy_net.state_dict()
        self.best_avg_score = -float('inf')
        self.episode_scores = deque(maxlen=100)
        self.kill_death_ratios = deque(maxlen=100)
        
        self.memory_monitor = MemoryMonitor(logger, config.memory_warning_threshold)
        
    def select_action(self, state: torch.Tensor) -> int:
        self.total_steps += 1
        
        decay_ratio = min(1.0, self.total_steps / self.config.epsilon_decay_steps)
        decay_factor = decay_ratio ** self.config.epsilon_power
        self.epsilon = self.config.epsilon_start - (self.config.epsilon_start - self.config.epsilon_end) * decay_factor
        
        if random.random() < self.epsilon and not self.config.use_noisy_nets:
            return random.randrange(self.action_space_size)
            
        with torch.no_grad():
            q_values = self.policy_net(state.unsqueeze(0))
            action = q_values.argmax().item()
            
        return action
        
    def learn(self) -> Optional[float]:
        if len(self.memory) < self.config.min_replay_size:
            return None
            
        tree_idx, samples, weights = self.memory.sample(self.config.batch_size)
        if samples is None:
            return None
            
        batch_data = [exp.decompress(self.config.device) for exp in samples]
        states, actions, rewards, next_states, dones = zip(*batch_data)
        
        states = torch.stack(states)
        actions = torch.tensor(actions, device=self.config.device, dtype=torch.long).view(-1, 1)
        rewards = torch.tensor(rewards, device=self.config.device, dtype=torch.float32).view(-1, 1)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, device=self.config.device, dtype=torch.float32).view(-1, 1)
        weights = weights.to(self.config.device).view(-1, 1)
        
        if self.config.clip_rewards:
            rewards = torch.clamp(rewards, -1, 1)
            
        q_current = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            if self.config.use_double_dqn:
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                q_next = self.target_net(next_states).gather(1, next_actions)
            else:
                q_next = self.target_net(next_states).max(dim=1, keepdim=True)[0]
                
            q_target = rewards + (self.config.gamma ** self.config.n_steps) * q_next * (1 - dones)
            
        td_errors = (q_target - q_current).abs().detach().cpu().numpy().flatten()
        self.memory.update_priorities(tree_idx, td_errors)
        
        loss = (weights * F.smooth_l1_loss(q_current, q_target, reduction='none')).mean()
        
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.gradient_clip_value)
        
        self.optimizer.step()
        
        if self.total_steps % 1000 == 0:
            self.scheduler.step()
            
        if self.config.use_noisy_nets:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()
            
        if self.total_steps % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.logger.info(f"Target network updated at step {self.total_steps}")
            
        return loss.item()
        
    def train(self, game, actions):
        for episode in range(1, self.config.episodes + 1):
            if episode % self.config.gc_interval == 0:
                gc.collect()
                self.logger.debug("Garbage collection performed")
                
            if episode % self.config.clear_cuda_cache_interval == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.debug("CUDA cache cleared")
                
            memory_percent = self.memory_monitor.check_memory()
            
            game.new_episode()
            
            initial_stats = {v.name: game.get_game_variable(v) for v in game.get_available_game_variables()}
            last_vars = initial_stats.copy()
            
            episode_score = 0
            episode_steps = 0
            episode_losses = []
            
            while not game.is_episode_finished():
                state = game.get_state()
                if state is None:
                    break
                    
                processed_state = preprocess_frame(state, self.config.device, self.config)
                
                action_idx = self.select_action(processed_state)
                
                game.make_action(actions[action_idx], self.config.frame_skip)
                episode_steps += 1
                
                current_vars = {v.name: game.get_game_variable(v) for v in game.get_available_game_variables()}
                
                rwd_hit = 2.0 * (current_vars["HITCOUNT"] - last_vars["HITCOUNT"])
                rwd_hit_taken = -0.1 * (current_vars["HITS_TAKEN"] - last_vars["HITS_TAKEN"])
                rwd_frag = 100.0 * (current_vars["FRAGCOUNT"] - last_vars["FRAGCOUNT"])
                rwd_death = -100.0 if current_vars["DEATHCOUNT"] > last_vars["DEATHCOUNT"] else 0.0
                rwd_activity = 0.01 if action_idx in [1, 2] else 0
                
                reward = rwd_hit + rwd_hit_taken + rwd_frag + rwd_death + rwd_activity
                episode_score += reward
                
                last_vars = current_vars
                
                self.memory.push(processed_state, action_idx, reward, game.is_episode_finished())
                
                if self.total_steps > self.config.min_replay_size and self.total_steps % self.config.train_freq == 0:
                    loss = self.learn()
                    if loss is not None:
                        episode_losses.append(loss)
                        
                del processed_state
                
            final_stats = {v.name: game.get_game_variable(v) for v in game.get_available_game_variables()}
            
            frags = final_stats["FRAGCOUNT"] - initial_stats["FRAGCOUNT"]
            deaths = final_stats["DEATHCOUNT"] - initial_stats["DEATHCOUNT"]
            hits_given = final_stats["HITCOUNT"] - initial_stats["HITCOUNT"]
            hits_taken = final_stats["HITS_TAKEN"] - initial_stats["HITS_TAKEN"]
            kd_ratio = frags / max(deaths, 1)
            
            self.episode_scores.append(episode_score)
            self.kill_death_ratios.append(kd_ratio)
            
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            
            self.logger.info(
                f"EP {episode:<5d} | Score:{episode_score:8.2f} | K/D:{frags}/{deaths} ({kd_ratio:4.2f}) | "
                f"Hits(G/T):{hits_given:3.0f}/{hits_taken:3.0f} | Loss:{avg_loss:7.4f} | "
                f"Eps:{self.epsilon:.4f} | LR:{self.optimizer.param_groups[0]['lr']:.6f} | "
                f"Mem:{memory_percent:.1f}%"
            )
            
            if episode % 100 == 0 and len(self.episode_scores) == 100:
                avg_score = np.mean(self.episode_scores)
                avg_kd = np.mean(self.kill_death_ratios)
                
                self.logger.info("=" * 100)
                self.logger.info(f"****** SUMMARY: EPISODE {episode} (Steps: {self.total_steps}) ******")
                self.logger.info(f"Avg Score (100ep): {avg_score:.2f} | Avg K/D: {avg_kd:.3f}")
                self.logger.info(f"Replay Buffer Size: {len(self.memory)} | Memory Usage: {memory_percent:.1f}%")
                self.logger.info("=" * 100)
                
                if avg_score > self.best_avg_score:
                    self.best_avg_score = avg_score
                    self.best_model_state_dict = self.policy_net.state_dict().copy()
                    self.logger.info(f"*** NEW BEST AVG SCORE: {self.best_avg_score:.2f}! ***")
                    
                    torch.save({
                        'model_state_dict': self.best_model_state_dict,
                        'config': asdict(self.config),
                        'episode': episode,
                        'avg_score': avg_score,
                        'total_steps': self.total_steps
                    }, 'best_model_checkpoint.pth')
                    
                if episode % 500 == 0:
                    torch.save({
                        'model_state_dict': self.policy_net.state_dict(),
                        'target_state_dict': self.target_net.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'config': asdict(self.config),
                        'episode': episode,
                        'total_steps': self.total_steps,
                        'epsilon': self.epsilon,
                        'memory_size': len(self.memory)
                    }, f'checkpoint_ep{episode}.pth')
                    self.logger.info(f"Checkpoint saved: checkpoint_ep{episode}.pth")

def setup_training(config: TrainingConfig) -> logging.Logger:
    os.makedirs('logs', exist_ok=True)
    
    log_filename = f"logs/training_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    print(f"Starting training. Logs: {log_filename}")
    
    logger = logging.getLogger('DoomAI')
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(console_handler)
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    logger.info(f"Device: {config.device.upper()} | PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
    logger.info(f"System RAM: {psutil.virtual_memory().total / 1024**3:.2f} GB")
    logger.info(f"CPU Cores: {psutil.cpu_count()}")
    
    return logger

def export_to_onnx(model_state_dict, config: TrainingConfig, action_space_size: int, logger):
    try:
        logger.info("Exporting model to ONNX...")
        
        input_channels = 1
        if config.use_depth:
            input_channels += 1
            
        model = DuelingDQN(input_channels, action_space_size, config.use_noisy_nets).to(config.device)
        model.load_state_dict(model_state_dict)
        model.eval()
        
        if config.use_noisy_nets:
            for module in model.modules():
                if isinstance(module, NoisyLinear):
                    module.training = False
                    
        dummy_input = torch.randn(1, input_channels, 128, 128).to(config.device)
        
        output_path = "doom_agent.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=11,
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            verbose=False
        )
        
        import onnx
        onnx_model = onnx.load(output_path)
        meta = onnx_model.metadata_props.add()
        meta.key = "config"
        meta.value = json.dumps(config.to_json_dict())
        onnx.save(onnx_model, output_path)
        
        logger.info(f"ONNX export successful: {output_path}")
        logger.info(f"Config metadata: {config.to_json_dict()}")
        
    except Exception as e:
        logger.error(f"ONNX export failed: {e}", exc_info=True)

def main():
    config = TrainingConfig()
    logger = setup_training(config)
    
    game = None
    try:
        game = create_doom_game(config)
        actions = create_action_space()
        
        assert len(actions) == 8, f"Action space must have 8 actions, got {len(actions)}"
        
        agent = Agent(config, len(actions), logger)
        
        logger.info("=" * 80)
        logger.info("DOOM AI TRAINING")
        logger.info("=" * 80)
        for key, value in asdict(config).items():
            logger.info(f"{key:30}: {value}")
        logger.info("=" * 80)
        logger.info(f"Action Space Size: {len(actions)} (0=no-op, 1-7=actions)")
        logger.info(f"Input Channels: {agent.input_channels} (screen + {'depth' if config.use_depth else ''})")
        logger.info("=" * 80)
        
        agent.train(game, actions)
        
        logger.info("Training complete. Exporting best model...")
        export_to_onnx(agent.best_model_state_dict, config, len(actions), logger)
                
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
        
    finally:
        if game:
            game.close()
        logger.info("Training session ended.")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()